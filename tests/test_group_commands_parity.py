# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""Group, component and paint commands the older tests never drive on
their own — Explode View, Reparent, Group in Context, Rename, and the
face-paint commands on loose and grouped faces — each followed through undo
and redo. They are also the golden scripts that hold the Rust port of the
same commands (M3.10) to Python's arithmetic: matrices with rotation, so
the Qt type flags matter."""
from __future__ import annotations

from PySide6.QtGui import QMatrix4x4, QVector3D as V

from core.group import Group
from core.history import (ExplodeViewCommand, GroupInContextCommand, History,
                          MakeGroupCommand, MakeNestedGroupCommand,
                          RenameGroupCommand, ReparentGroupsCommand,
                          SetFaceBackCommand, SetFaceColorCommand,
                          SetFaceMaterialTagCommand, SetFaceOpacityCommand,
                          SetFaceTextureCommand)
from core.materials import Material
from core.mesh import Mesh
from core.scene import Scene


def _quad(x0=0.0, y0=0.0, z=0.0, size=1.0):
    m = Mesh()
    m.add_face([V(x0, y0, z), V(x0 + size, y0, z), V(x0 + size, y0 + size, z),
                V(x0, y0 + size, z)])
    return m


def _part(name, x, y, z, turn=0.0, size=1.0):
    g = Group(_quad(size=size), name=name)
    xf = QMatrix4x4()
    xf.translate(x, y, z)
    if turn:
        xf.rotate(turn, 0.0, 0.0, 1.0)
    g.xform = xf
    return g


def _container(name, parts, turn=0.0):
    c = Group(Mesh(), name=name)
    c.adopt(parts)
    c.component = False
    if turn:
        xf = QMatrix4x4()
        xf.translate(1.5, -2.0, 0.25)
        xf.rotate(turn, 0.0, 1.0, 0.0)
        c.xform = xf
    return c


def _round_trip(history, steps):
    for _ in range(steps):
        assert history.undo()
    for _ in range(steps):
        assert history.redo()


def test_explode_view_pulls_the_parts_apart_and_back():
    scene = Scene()
    box = _container("Banca", [
        _part("Asiento", 0.0, 0.0, 0.5, turn=0.0),
        _part("Pata 1", -1.0, -1.0, 0.0, turn=30.0, size=0.3),
        _part("Pata 2", 1.7, -1.1, 0.0, turn=-45.0, size=0.3),
        _part("Respaldo", 0.3, 1.2, 0.9, turn=12.5, size=0.7),
    ])
    scene.groups.append(box)
    h = History(scene)
    h.execute(ExplodeViewCommand(box, 0.5))
    h.execute(ExplodeViewCommand(box, 1.25, "z"))
    h.execute(ExplodeViewCommand(box, 0.0))
    assert box.exploded is None
    _round_trip(h, 3)


def test_reparent_keeps_the_world_place():
    scene = Scene()
    a = _part("A", 4.0, 1.0, 0.0, turn=20.0)
    c = Group(_quad(2.0, 2.0), name="C")                  # classic: baked
    b = _container("B", [_part("Dentro", 0.5, 0.5, 0.0)], turn=33.0)
    d = _container("D", [_part("Otro", 1.0, 0.0, 0.0)])
    scene.groups.extend([a, c, b, d])
    h = History(scene)
    h.execute(ReparentGroupsCommand([a, c], b))
    h.execute(ReparentGroupsCommand([a], None))
    h.execute(ReparentGroupsCommand([b.children[0]], d))
    _round_trip(h, 3)


def test_group_in_context_makes_the_group_inside_the_open_one():
    scene = Scene()
    c1 = _part("Uno", 0.0, 0.0, 0.0)
    c2 = _part("Dos", 3.0, 0.0, 0.0, turn=15.0)
    ctx = _container("Abierto", [c1, c2], turn=10.0)
    solo = Group(Mesh(), name="Vacio")
    scene.groups.extend([ctx, solo])
    scene.mesh.add_face([V(10, 10, 0), V(12, 10, 0), V(12, 12, 0), V(10, 12, 0)])
    scene.mesh.add_face([V(20, 10, 0), V(22, 10, 0), V(22, 12, 0), V(20, 12, 0)])
    h = History(scene)
    faces = list(scene.mesh.faces)
    h.execute(GroupInContextCommand(
        MakeNestedGroupCommand([], [], [c1, c2], name="Par"), ctx))
    h.execute(GroupInContextCommand(
        MakeGroupCommand([faces[0]], []), solo))
    assert solo.xform is not None and solo.component is False
    _round_trip(h, 2)


def test_paint_commands_on_loose_and_grouped_faces():
    scene = Scene()
    scene.mesh.add_face([V(0, 0, 0), V(2, 0, 0), V(2, 2, 0), V(0, 2, 0)])
    g = Group(_quad(5.0, 5.0, size=2.0), name="Caja")
    scene.groups.append(g)
    loose = list(scene.mesh.faces)
    inside = list(g.mesh.faces)
    both = loose + inside
    h = History(scene)
    h.execute(SetFaceColorCommand(both, (0.2, 0.4, 0.6)))
    h.execute(SetFaceOpacityCommand(both, 0.5))
    h.execute(SetFaceMaterialTagCommand(
        loose, "Azul", Material("Azul", color=(0.2, 0.4, 0.6))))
    h.execute(SetFaceTextureCommand(inside, {"path": "x.png", "sw": 1.0, "sh": 1.0}))
    h.execute(SetFaceBackCommand(both, {"color": (0.0, 1.0, 0.0), "mat": "Verde"}))
    h.execute(SetFaceColorCommand(loose, None))
    h.execute(SetFaceOpacityCommand(inside, None))
    h.execute(RenameGroupCommand(g, "Cajón"))
    _round_trip(h, 8)
    assert g.name == "Cajón"
