# SPDX-License-Identifier: GPL-3.0-or-later
"""Command arguments to JSON and back (golden/SCHEMA.md, sections 3–4).

Plain JSON values stay as they are. Everything else is a one-key object
whose key starts with ``$``:

- values: ``$v`` (QVector3D), ``$m`` (QMatrix4x4), ``$color``, ``$t``
  (tuple), ``$set``, ``$dict`` (a dict whose keys are not all plain
  strings), ``$obj`` (a scene object by value: its ``to_dict()``);
- selectors of live objects, resolved by position when the step runs:
  ``$face``, ``$edge``, ``$vertex``, ``$group``, ``$kept``, ``$mesh``,
  ``$ref`` (an annotation, guide, plane… by its index in the scene);
- ``$cmd``: another command, as its own op.

A selector must name exactly one object; otherwise the value cannot be
written and the test is not harvested (``NotSerializable`` says why).
"""
from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path
from typing import Any

#: The repository root, and the corpus' own copies of external files.
REPO = Path(__file__).resolve().parents[2]
FILES = "golden/files"
_FILE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def vendor_file(s: str) -> str:
    """An absolute path to an existing image outside the repository (a test's
    ``tmp_path``) becomes ``golden/files/<sha>-<name>``, a copy kept in the
    corpus: a script must replay on a machine that has never seen that
    folder. Anything else is returned as it is."""
    if len(s) < 4 or Path(s).suffix.lower() not in _FILE_EXT:
        return s
    try:
        p = Path(s)
        if not p.is_absolute() or not p.is_file():
            return s
        p = p.resolve()
        if REPO in p.parents:
            return s
        data = p.read_bytes()
    except OSError:
        return s
    name = f"{hashlib.sha256(data).hexdigest()[:12]}-{p.name}"
    dest = REPO / FILES / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    return f"{FILES}/{name}"


def resolve_file(s: str) -> str:
    """The inverse on replay: ``golden/files/…`` is a path in the repository."""
    if s.startswith(FILES + "/"):
        return str(REPO / s)
    return s


class NotSerializable(Exception):
    """This argument has no golden form (yet)."""


# Scene lists an argument may point into, and the class of their items.
SCENE_LISTS = ("dimensions", "text_labels", "guides", "section_planes",
               "image_planes", "geo_paths", "geo_points", "saved_views",
               "compositions", "layers")


def _point(p) -> list:
    return [p.x(), p.y(), p.z()]


# ---- where a live object is -------------------------------------------------

def group_paths(scene):
    """``(path, group)`` for every group of the scene tree, depth first."""
    stack = [((i,), g) for i, g in enumerate(scene.groups or ())]
    stack.reverse()
    while stack:
        path, g = stack.pop()
        yield path, g
        kids = list(enumerate(g.children or ()))
        for i, c in reversed(kids):
            stack.append((path + (i,), c))


def loose_mesh(scene):
    return getattr(scene, "_loose_mesh", None) or scene.mesh


def meshes(scene):
    """``(path or None, mesh)``: the loose mesh, then each distinct group
    mesh at its first group."""
    seen = set()
    loose = loose_mesh(scene)
    seen.add(id(loose))
    yield None, loose
    for path, g in group_paths(scene):
        if id(g.mesh) not in seen:
            seen.add(id(g.mesh))
            yield list(path), g.mesh


def _mesh_at(scene, path):
    if path is None:
        return loose_mesh(scene)
    return group_at(scene, path).mesh


def group_at(scene, path):
    items = scene.groups
    g = None
    for i in path:
        g = items[i]
        items = g.children or []
    if g is None:
        raise NotSerializable("empty group path")
    return g


def _loop_set(face) -> frozenset:
    return frozenset(tuple(_point(v.position)) for v in face.loop)


def _edge_set(e) -> frozenset:
    return frozenset((tuple(_point(e.v0.position)), tuple(_point(e.v1.position))))


# ---- encoding ----------------------------------------------------------------

class Encoder:
    """Encodes against ``scene`` as it is now."""

    def __init__(self, scene, command_ops=None) -> None:
        self.scene = scene
        # Inner commands already run, by id: their recorded ops.
        self.command_ops = command_ops if command_ops is not None else {}

    def encode(self, v: Any):
        from PySide6.QtGui import QColor, QMatrix4x4, QVector3D

        if isinstance(v, str):
            return vendor_file(v)
        if v is None or isinstance(v, bool):
            return v
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            if not math.isfinite(v):
                raise NotSerializable("non-finite float")
            return v
        if isinstance(v, QVector3D):
            return {"$v": _point(v)}
        if isinstance(v, QMatrix4x4):
            return {"$m": list(v.data())}
        if isinstance(v, QColor):
            return {"$color": v.name(QColor.NameFormat.HexArgb)}
        if isinstance(v, list):
            return [self.encode(x) for x in v]
        if isinstance(v, tuple):
            return {"$t": [self.encode(x) for x in v]}
        if isinstance(v, (set, frozenset)):
            items = [self.encode(x) for x in v]
            import json
            return {"$set": sorted(items, key=lambda x: json.dumps(x, sort_keys=True))}
        if isinstance(v, dict):
            if all(isinstance(k, str) and not k.startswith("$") for k in v):
                return {k: self.encode(x) for k, x in v.items()}
            return {"$dict": [[self.encode(k), self.encode(x)] for k, x in v.items()]}
        from core.history import Command
        if isinstance(v, Command):
            op = self.command_ops.get(id(v))
            if op is None:
                raise NotSerializable(f"inner {type(v).__name__} never ran")
            return {"$cmd": op}
        return self._live(v)

    def _live(self, v):
        from core.group import Group
        from core.mesh import Edge, Face, Mesh, Vertex

        scene = self.scene
        if isinstance(v, Face):
            for path, mesh in meshes(scene):
                if v in mesh.faces:
                    key = _loop_set(v)
                    if sum(1 for f in mesh.faces if _loop_set(f) == key) != 1:
                        raise NotSerializable("two faces on the same loop")
                    return {"$face": {"loop": [_point(x.position) for x in v.loop], "in": path}}
            raise NotSerializable("a face outside the scene")
        if isinstance(v, Edge):
            for path, mesh in meshes(scene):
                if v in mesh.edges:
                    key = _edge_set(v)
                    if sum(1 for e in mesh.edges if _edge_set(e) == key) != 1:
                        raise NotSerializable("two edges on the same ends")
                    return {"$edge": {"ends": [_point(v.v0.position), _point(v.v1.position)], "in": path}}
            raise NotSerializable("an edge outside the scene")
        if isinstance(v, Vertex):
            for path, mesh in meshes(scene):
                if any(x is v for x in mesh.vertices):
                    return {"$vertex": {"at": _point(v.position), "in": path}}
            raise NotSerializable("a vertex outside the scene")
        if isinstance(v, Group):
            for path, g in group_paths(scene):
                if g is v:
                    return {"$group": list(path)}
            for i, g in enumerate(getattr(scene, "kept_components", None) or ()):
                if g is v:
                    return {"$kept": i}
            raise NotSerializable("a group outside the scene")
        if isinstance(v, Mesh):
            for path, mesh in meshes(scene):
                if mesh is v:
                    return {"$mesh": path}
            raise NotSerializable("a mesh outside the scene")
        for name in SCENE_LISTS:
            items = getattr(scene, name, None) or ()
            for i, x in enumerate(items):
                if x is v:
                    return {"$ref": [name, i]}
        return self._by_value(v)

    def _by_value(self, v):
        from core import geometry
        from core.dimension import Dimension

        # core.geometry's Face and Edge are values (positions), not mesh
        # entities: DeleteFaceCommand finds the mesh face by this loop.
        if isinstance(v, geometry.Face):
            return {"$obj": "geometry.Face", "data": {
                "vertices": [_point(p) for p in v.vertices],
                "holes": [[_point(p) for p in h] for h in v.holes],
                "attrs": self.encode(v.attrs)}}
        if isinstance(v, geometry.Edge):
            return {"$obj": "geometry.Edge", "data": {"a": _point(v.a), "b": _point(v.b)}}

        if isinstance(v, Dimension):
            return {"$obj": "Dimension", "data": {
                "a": _point(v.a), "b": _point(v.b), "offset": _point(v.offset),
                "layer": v.layer, "text": v.text, "axis": v.axis}}
        cls = type(v)
        if callable(getattr(v, "to_dict", None)) and callable(getattr(cls, "from_dict", None)):
            return {"$obj": f"{cls.__module__}.{cls.__qualname__}",
                    "data": self.encode(v.to_dict())}
        raise NotSerializable(f"{cls.__module__}.{cls.__qualname__}")


# ---- decoding ----------------------------------------------------------------

class Decoder:
    """Decodes against ``scene`` as it is now (the mirror of Encoder)."""

    def __init__(self, scene, make_command, ran=None) -> None:
        self.scene = scene
        self.make_command = make_command
        #: Commands already built in this step, by their op's ``rid``.
        self.ran = ran if ran is not None else {}

    def decode(self, v):
        from PySide6.QtGui import QColor, QMatrix4x4, QVector3D

        if isinstance(v, list):
            return [self.decode(x) for x in v]
        if isinstance(v, str):
            return resolve_file(v)
        if not isinstance(v, dict):
            return v
        if len(v) != 1 or not next(iter(v)).startswith("$"):
            if set(v) == {"$obj", "data"}:
                return self._obj(v["$obj"], v["data"])
            return {k: self.decode(x) for k, x in v.items()}
        (tag, x), = v.items()
        scene = self.scene
        if tag == "$v":
            return QVector3D(*x)
        if tag == "$m":
            m = QMatrix4x4(*x)
            return m.transposed()            # data() is column major
        if tag == "$color":
            return QColor(x)
        if tag == "$t":
            return tuple(self.decode(i) for i in x)
        if tag == "$set":
            return {self.decode(i) for i in x}
        if tag == "$dict":
            return {self._key(self.decode(k)): self.decode(i) for k, i in x}
        if tag == "$cmd":
            return self.make_command(x)
        if tag == "$ran":
            return self.ran[x]
        if tag == "$face":
            mesh = _mesh_at(scene, x["in"])
            key = frozenset(tuple(p) for p in x["loop"])
            hits = [f for f in mesh.faces if _loop_set(f) == key]
            if len(hits) != 1:
                raise LookupError(f"$face matched {len(hits)} faces")
            return hits[0]
        if tag == "$edge":
            mesh = _mesh_at(scene, x["in"])
            key = frozenset(tuple(p) for p in x["ends"])
            hits = [e for e in mesh.edges if _edge_set(e) == key]
            if len(hits) != 1:
                raise LookupError(f"$edge matched {len(hits)} edges")
            return hits[0]
        if tag == "$vertex":
            mesh = _mesh_at(scene, x["in"])
            hits = [p for p in mesh.vertices if _point(p.position) == x["at"]]
            if len(hits) != 1:
                raise LookupError(f"$vertex matched {len(hits)} vertices")
            return hits[0]
        if tag == "$group":
            return group_at(scene, x)
        if tag == "$kept":
            return scene.kept_components[x]
        if tag == "$mesh":
            return _mesh_at(scene, x)
        if tag == "$ref":
            name, i = x
            return getattr(scene, name)[i]
        raise ValueError(f"unknown tag {tag}")

    @staticmethod
    def _key(k):
        return k

    def _obj(self, name, data):
        from PySide6.QtGui import QVector3D

        if name == "geometry.Face":
            from core import geometry
            return geometry.Face([QVector3D(*p) for p in data["vertices"]],
                                 [[QVector3D(*p) for p in h] for h in data["holes"]],
                                 self.decode(data["attrs"]))
        if name == "geometry.Edge":
            from core import geometry
            return geometry.Edge(QVector3D(*data["a"]), QVector3D(*data["b"]))
        if name == "Dimension":
            from core.dimension import Dimension
            return Dimension(QVector3D(*data["a"]), QVector3D(*data["b"]),
                             QVector3D(*data["offset"]), layer=data["layer"],
                             text=data["text"], axis=data["axis"])
        import importlib
        mod, _, qual = name.rpartition(".")
        cls = importlib.import_module(mod)
        for part in qual.split("."):
            cls = getattr(cls, part)
        return cls.from_dict(self.decode(data))
